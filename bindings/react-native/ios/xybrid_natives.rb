# Resolves the two iOS pieces the npm tarball does not carry by default, at
# `pod install` time (the podspec calls this while CocoaPods evaluates it):
#
#   ios/Frameworks/XybridFFI.xcframework  — the Rust core (~64 MB zipped).
#     Downloaded from the GitHub Release for this package's version and
#     checked against the SHA-256 in package.json, exactly like Package.swift's
#     binaryTarget. Cached in ~/.xybrid/cache/xcframework, next to the SDK's
#     other caches, so each version downloads once per machine.
#   ios/XybridSwift/{Xybrid,xybrid_bolt}.swift — the Swift SDK sources. The
#     published tarball includes them; in the xybrid monorepo they are synced
#     from bindings/apple on every install.
#
# Where the XCFramework comes from, first match wins:
#   1. XYBRID_XCFRAMEWORK_PATH — an XybridFFI.xcframework directory or .zip
#   2. an ios/Frameworks/XybridFFI.xcframework you staged yourself
#   3. in the monorepo: <repo>/bazel-bin/bindings/apple/XybridFFI.xcframework.zip
#      (a release download could not match the working tree's C ABI, so the
#      monorepo never falls back to one)
#   4. otherwise: the pinned release asset (XYBRID_NATIVES_BASE_URL = mirror)
#
# Plain Ruby + curl/unzip/shasum from macOS: this runs inside `pod install`,
# so it must not need anything a stock Mac lacks.

require 'digest'
require 'fileutils'
require 'json'
require 'tmpdir'

module XybridNatives
  FRAMEWORK = 'XybridFFI.xcframework'.freeze
  STAMP = '.xybrid-natives'.freeze
  SWIFT_SOURCES = %w[Xybrid.swift xybrid_bolt.swift].freeze
  DEFAULT_BASE_URL = 'https://github.com/xybrid-ai/xybrid/releases/download'.freeze

  module_function

  # Make sure every native input exists; returns the framework path relative
  # to the pod root, for `vendored_frameworks`.
  def prepare!(root)
    # Resolve symlinks: `file:` installs link node_modules to the source tree.
    root = File.realpath(root)
    package = JSON.parse(File.read(File.join(root, 'package.json')))
    monorepo = File.expand_path('../..', root) if File.file?(File.join(root, '../apple/Sources/Xybrid/Xybrid.swift'))
    ensure_swift_sources!(root, monorepo)
    ensure_xcframework!(root, package, monorepo)
    "ios/Frameworks/#{FRAMEWORK}"
  end

  def log(message)
    if defined?(Pod::UI)
      Pod::UI.puts("[react-native-xybrid] #{message}")
    else
      puts("[react-native-xybrid] #{message}")
    end
  end

  def fail!(message)
    raise(defined?(Pod::Informative) ? Pod::Informative : RuntimeError, "[react-native-xybrid] #{message}")
  end

  def ensure_swift_sources!(root, monorepo)
    staged = File.join(root, 'ios', 'XybridSwift')
    if monorepo
      # Always sync, so a stale copy can never drift from bindings/apple.
      source = File.join(monorepo, 'bindings', 'apple', 'Sources', 'Xybrid')
      FileUtils.mkdir_p(staged)
      SWIFT_SOURCES.each do |name|
        from = File.join(source, name)
        to = File.join(staged, name)
        FileUtils.cp(from, to) unless File.file?(to) && FileUtils.identical?(from, to)
      end
      return
    end
    return if SWIFT_SOURCES.all? { |name| File.file?(File.join(staged, name)) }

    fail!('ios/XybridSwift is missing the Swift SDK sources; reinstall react-native-xybrid from npm.')
  end

  def ensure_xcframework!(root, package, monorepo)
    frameworks = File.join(root, 'ios', 'Frameworks')
    target = File.join(frameworks, FRAMEWORK)
    stamp = File.join(frameworks, STAMP)
    installed = File.file?(stamp) ? File.read(stamp).strip : nil
    version = package.fetch('version')

    local = ENV['XYBRID_XCFRAMEWORK_PATH'].to_s.strip
    local = File.expand_path(local) unless local.empty?
    # A framework staged by hand (no stamp) is used as-is.
    return if local.empty? && File.directory?(target) && installed.nil?

    if local.empty? && monorepo
      local = File.join(monorepo, 'bazel-bin', 'bindings', 'apple', 'XybridFFI.xcframework.zip')
      unless File.exist?(local)
        fail!('build the XCFramework first: `bazel build --config=ios //bindings/apple:XybridFFI` ' \
              '(or set XYBRID_XCFRAMEWORK_PATH).')
      end
    end

    unless local.empty?
      fail!("XYBRID_XCFRAMEWORK_PATH does not exist: #{local}") unless File.exist?(local)
      source = "local:#{local}:#{fingerprint(local)}"
      return if File.directory?(target) && installed == source

      replace!(frameworks, target) do
        File.directory?(local) ? FileUtils.cp_r(local, target) : unzip!(local, frameworks)
      end
      File.write(stamp, "#{source}\n")
      log("Using the XCFramework from #{local}")
      return
    end

    sha256 = package.dig('xybrid', 'iosXcframeworkSha256').to_s.strip
    fail!("no XCFramework checksum is pinned for #{version}; set XYBRID_XCFRAMEWORK_PATH.") if sha256.empty?
    source = "release:#{version}:#{sha256}"
    return if File.directory?(target) && installed == source

    zip = cached_download!(version, sha256)
    replace!(frameworks, target) { unzip!(zip, frameworks) }
    File.write(stamp, "#{source}\n")
  end

  def replace!(frameworks, target)
    FileUtils.rm_rf(target)
    FileUtils.rm_f(File.join(frameworks, STAMP))
    FileUtils.mkdir_p(frameworks)
    yield
  end

  # Identifies a local XCFramework (directory or .zip) by every file's path,
  # size and mtime to the nanosecond. A directory's own mtime only moves when
  # its direct entries change, so rebuilding a library inside the framework
  # would otherwise keep serving the stale copy.
  def fingerprint(path)
    files = File.directory?(path) ? Dir.glob(File.join(path, '**', '*'), File::FNM_DOTMATCH) : [path]
    digest = Digest::SHA256.new
    files.select { |file| File.file?(file) }.sort.each do |file|
      stat = File.stat(file)
      digest << "#{file.delete_prefix(path)}\0#{stat.size}\0#{stat.mtime.to_i}.#{stat.mtime.nsec}\n"
    end
    digest.hexdigest
  end

  def cached_download!(version, sha256)
    cache = File.join(Dir.home, '.xybrid', 'cache', 'xcframework')
    name = "XybridFFI-v#{version}.xcframework.zip"
    zip = File.join(cache, name)
    return zip if File.file?(zip) && Digest::SHA256.file(zip).hexdigest == sha256

    base = ENV['XYBRID_NATIVES_BASE_URL'].to_s.strip
    base = DEFAULT_BASE_URL if base.empty?
    url = "#{base.chomp('/')}/v#{version}/#{name}"
    FileUtils.mkdir_p(cache)
    partial = "#{zip}.partial"
    log("Downloading #{url}")
    unless system('curl', '--fail', '--location', '--silent', '--show-error',
                  '--retry', '3', '--output', partial, url)
      FileUtils.rm_f(partial)
      fail!("could not download #{url}. Set XYBRID_NATIVES_BASE_URL to a mirror, or " \
            'XYBRID_XCFRAMEWORK_PATH to a local copy.')
    end
    actual = Digest::SHA256.file(partial).hexdigest
    unless actual == sha256
      FileUtils.rm_f(partial)
      fail!("checksum mismatch for #{name}: expected #{sha256}, got #{actual}")
    end
    FileUtils.mv(partial, zip)
    zip
  end

  def unzip!(zip, destination)
    Dir.mktmpdir('xybrid-xcframework') do |scratch|
      fail!("could not unzip #{zip}") unless system('unzip', '-q', '-o', zip, '-d', scratch)
      extracted = Dir.glob(File.join(scratch, '**', FRAMEWORK)).min_by(&:length)
      fail!("#{zip} does not contain #{FRAMEWORK}") unless extracted
      FileUtils.mv(extracted, File.join(destination, FRAMEWORK))
    end
  end
end
