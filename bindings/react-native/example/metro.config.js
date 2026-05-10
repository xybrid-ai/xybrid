const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

// The example resolves `react-native-xybrid` from the parent directory rather
// than via npm. Two pieces matter for that to work:
//   1. `watchFolders` lets Metro see file changes in ../src.
//   2. `resolver.nodeModulesPaths` makes the example's react/react-native deps
//      win over any duplicates that might exist in the parent — without this,
//      Metro can hand React two copies of itself and trip the invariant
//      ("Invalid hook call. Hooks can only be called inside…").

const projectRoot = __dirname;
const packageRoot = path.resolve(projectRoot, '..');

const config = {
  watchFolders: [packageRoot],
  resolver: {
    nodeModulesPaths: [
      path.resolve(projectRoot, 'node_modules'),
      path.resolve(packageRoot, 'node_modules'),
    ],
    // Force-resolve the parent's TS source through the example's react copy.
    extraNodeModules: {
      react: path.resolve(projectRoot, 'node_modules/react'),
      'react-native': path.resolve(projectRoot, 'node_modules/react-native'),
    },
  },
};

module.exports = mergeConfig(getDefaultConfig(projectRoot), config);
