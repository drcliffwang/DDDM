#!/bin/bash
# ============================================
# DDDM Production Docker Build
# ============================================
# Usage:
#   ./build-run.sh              - Build and run in foreground
#   ./build-run.sh background   - Run in background (detached)
#   ./build-run.sh build-only   - Build images without running
# ============================================

case "$1" in
  background)
    echo "🐳 Building DDDM Production Containers..."
    docker-compose build
    
    echo ""
    echo "🚀 Starting containers in BACKGROUND mode..."
    docker-compose up -d
    
    echo ""
    echo "✅ Containers running in background!"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    echo ""
    echo "💡 Tips:"
    echo "   View logs:    ./dev-run.sh logs"
    echo "   Stop:         ./dev-run.sh stop"
    echo "   Check status: ./dev-run.sh status"
    ;;
    
  build-only)
    echo "🔨 Building Docker images only (not running)..."
    docker-compose build
    echo ""
    echo "✅ Images built successfully!"
    echo "   Run './build-run.sh' to start containers."
    ;;
    
  *)
    echo "🐳 DDDM Production Build & Run"
    echo "=============================="
    echo ""
    echo "🔨 Building production containers..."
    docker-compose build
    
    echo ""
    echo "🚀 Starting containers..."
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    echo ""
    echo "📝 Press Ctrl+C to stop containers"
    echo ""
    
    docker-compose up
    ;;
esac
