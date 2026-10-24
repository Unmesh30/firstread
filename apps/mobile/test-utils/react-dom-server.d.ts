// The one react-dom/server API the hydration test uses (no @types/react-dom needed).
declare module 'react-dom/server' {
  export function renderToString(node: import('react').ReactNode): string;
}
