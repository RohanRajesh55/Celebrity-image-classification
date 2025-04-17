from app import create_app

app = create_app()

if __name__ == '__main__':
    # Bind to all network interfaces so the container is accessible externally.
    app.run(host="0.0.0.0", port=5000, debug=True)