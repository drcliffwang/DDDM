#!/bin/bash
# ============================================
# DDDM Development Docker Script
# ============================================
# Usage:
#   ./dev-run.sh          - Build and run (see logs)
#   ./dev-run.sh stop     - Stop all containers
#   ./dev-run.sh logs     - View container logs
#   ./dev-run.sh clean    - Remove containers and images
#   ./dev-run.sh status   - Check if containers are running
# ============================================

case "$1" in
  stop)
    echo "🛑 Stopping DDDM containers..."
    docker-compose down
    echo "✅ Containers stopped."
    ;;
    
  logs)
    echo "📋 Showing container logs (Ctrl+C to exit)..."
    docker-compose logs -f
    ;;
    
  clean)
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down --rmi local --volumes
    echo "✅ Containers, images, and volumes removed."
    ;;
    
  status)
    echo "📊 Container Status:"
    docker-compose ps
    ;;
    
  *)
    echo "🐳 DDDM Docker Development Mode"
    echo "================================"
    echo ""
    echo "🔨 Building containers..."
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
