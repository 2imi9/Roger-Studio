# Roger Studio frontend — Vite + React 19 + Tailwind v4.
#
# Two-stage layout: the deps stage caches pnpm-lock.yaml installs so a
# code-only edit doesn't reinstall node_modules. The runtime stage is
# `vite dev` (not a static build) so changes hot-reload during local
# development. For production-quality static serving, run `pnpm build`
# and serve `dist/` from any HTTP server (nginx, caddy, etc.) — out of
# scope for this dev quickstart.

FROM node:22-bookworm AS deps
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@latest --activate
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:22-bookworm AS runtime
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@latest --activate
COPY --from=deps /app/node_modules ./node_modules
COPY . .

EXPOSE 3000
# Bind 0.0.0.0 so the container's port forwards out of the bridge net.
# --host alone defaults to "any IP", which is what we need for `docker
# compose up` to expose the dev server on the host's localhost:3000.
CMD ["pnpm", "dev", "--host", "0.0.0.0", "--port", "3000"]
