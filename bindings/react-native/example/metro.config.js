// Metro config for consuming the symlinked parent package (react-native-xybrid)
// from source in this monorepo. Two problems this solves:
//
//   1. The parent package's `react-native` field points Metro at its TS source
//      (../src), which lives outside this project — so Metro must *watch* the
//      parent dir or it can't read those files.
//   2. That source does `import 'react-native'`, but react-native is only
//      installed in THIS example's node_modules, not the parent's. Without a
//      mapping, Metro resolves `react-native` relative to the parent and fails
//      with "Unable to resolve module react-native". `extraNodeModules` pins
//      react-native / react (and the package itself) to single copies so the
//      symlinked source resolves them here.
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const projectRoot = __dirname;
const packageRoot = path.resolve(projectRoot, '..');

const config = getDefaultConfig(projectRoot);

// Watch the parent package so Metro picks up edits to ../src.
config.watchFolders = [packageRoot];

// Resolve deps from the example first, then the package.
config.resolver.nodeModulesPaths = [
  path.resolve(projectRoot, 'node_modules'),
  path.resolve(packageRoot, 'node_modules'),
];

// Single-copy pins: the symlinked package's source must use THIS example's
// react-native / react, or Metro can't find them (and a duplicate React would
// break hooks anyway).
config.resolver.extraNodeModules = {
  'react-native': path.resolve(projectRoot, 'node_modules/react-native'),
  react: path.resolve(projectRoot, 'node_modules/react'),
  'react-native-xybrid': packageRoot,
};

module.exports = config;
