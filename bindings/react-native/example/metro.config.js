// Metro config for running the example against the package source in ../src
// (the `react-native` field), so JS edits need no build step.
//
//   1. Metro must watch the package directory to read ../src at all.
//   2. `react` and `react-native` must resolve to THIS app's copies, even when
//      ../node_modules holds the package's own dev copies (after `npm install`
//      in bindings/react-native). Hierarchical lookup from ../src would find
//      those first and bundle a second React Native — whose
//      TurboModuleRegistry does not match the native runtime, so
//      `RNXybrid could not be found` (and duplicate-React hook errors).
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const projectRoot = __dirname;
const packageRoot = path.resolve(projectRoot, '..');
const singletons = ['react', 'react-native'];

const config = getDefaultConfig(projectRoot);

config.watchFolders = [packageRoot];
config.resolver.nodeModulesPaths = [path.resolve(projectRoot, 'node_modules')];
config.resolver.extraNodeModules = { '@xybrid/react-native': packageRoot };

const upstreamResolve = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  const isSingleton = singletons.some(
    (name) => moduleName === name || moduleName.startsWith(`${name}/`),
  );
  // Resolve singletons as if imported from the app root.
  const effective = isSingleton
    ? { ...context, originModulePath: path.join(projectRoot, 'index.ts') }
    : context;
  return (upstreamResolve ?? context.resolveRequest)(effective, moduleName, platform);
};

module.exports = config;
