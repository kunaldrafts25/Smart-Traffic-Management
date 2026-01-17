#!/usr/bin/env python3
import argparse
import sys
import logging
import subprocess
from pathlib import Path


def run_app(args):
    logging.info(f"Launching Streamlit on port {args.port}")    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(args.port),
            "--server.address", "localhost",
        ]
        if args.headless:
            cmd.append("--server.headless=true")
        
        subprocess.run(cmd, cwd=Path(__file__).parent)
        return 0
    except Exception as e:
        logging.error(f"Failed to launch app: {e}")
        print(f"Error: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="4-Lane Traffic Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py app                  Launch web interface
  python main.py app --port 8502      Launch on specific port
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # App command
    app_parser = subparsers.add_parser('app', help='Launch Streamlit web interface')
    app_parser.add_argument('--port', type=int, default=8501, help='Port number')
    app_parser.add_argument('--headless', action='store_true', help='Run in headless mode')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.command == 'app':
        return run_app(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
