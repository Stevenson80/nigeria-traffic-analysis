#!/usr/bin/env bash
# build.sh - Install system dependencies for WeasyPrint on Render

echo "Installing system dependencies for WeasyPrint..."

# Update package list and install required system libraries
apt-get update
apt-get install -y \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info

# Install fonts
apt-get install -y \
    fonts-dejavu \
    fonts-liberation \
    fonts-freefont-ttf

echo "System dependencies installed successfully!"