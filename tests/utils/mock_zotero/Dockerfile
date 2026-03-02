FROM node:25.2.1-bookworm-slim as builder
RUN apt-get update && apt-get install -y git git-lfs perl python3 curl wget zip unzip xz-utils rsync libgtk-3-0 libasound2 libx11-xcb1 && apt-get clean && rm -rf /var/lib/apt/lists/* 
RUN git clone --recursive -b enable_api https://github.com/TideDra/zotero.git /zotero

WORKDIR /zotero

RUN npm i && npm run build && app/scripts/dir_build -p l


FROM debian:bookworm-slim

COPY --from=builder /zotero/app/staging/Zotero_linux-x86_64/ /zotero
COPY ./zotero.sqlite /root/Zotero/zotero.sqlite
WORKDIR /zotero

# Install xvfb and required libraries including graphics libraries
RUN apt-get update && apt-get install -y \
    xvfb \
    libgtk-3-0 \
    libasound2 \
    libx11-xcb1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Environment variables to fix errors
ENV DISPLAY=:99


EXPOSE 23119


CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 & ./zotero"]