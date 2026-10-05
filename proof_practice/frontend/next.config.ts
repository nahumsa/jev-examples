import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  async rewrites() {
    // Browser uses same-origin requests; provider credentials never reach Next.js.
    const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};
export default config;
