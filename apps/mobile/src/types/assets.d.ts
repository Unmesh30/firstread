// Asset modules Metro bundles beyond Expo's own declarations (metro.config.js).
declare module '*.woff2' {
  /** The asset's module id, as for images and TTF fonts. */
  const asset: number;
  export default asset;
}
