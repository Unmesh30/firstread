// Expo's Metro config, plus .woff2 as an asset type (the web headline font; native
// platforms load the TTF, which Metro already bundles).
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('woff2');

module.exports = config;
