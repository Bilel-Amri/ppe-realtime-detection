# --------------------------------------------------------------------------- #
# Stage 1 - build the React application
# --------------------------------------------------------------------------- #
FROM node:20-alpine AS builder

WORKDIR /app

# Copy the manifest first so that the dependency layer is cached.
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund --legacy-peer-deps
COPY . .

# The production build talks to the backend through the nginx proxy, so the
# API base URL is a relative path ("/api") and there is no CORS involved.
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build


# --------------------------------------------------------------------------- #
# Stage 2 - serve the static bundle with nginx
# --------------------------------------------------------------------------- #
FROM nginx:1.27-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
