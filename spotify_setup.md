# Spotify Integration Setup

To use Spotify integration, you need to set up a Spotify Developer application and configure environment variables.

## 1. Create a Spotify Developer Application

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your Spotify account
3. Click "Create App"
4. Fill in the application details:
   - App name: "Music Visualization Viewer"
   - App description: "Web application for visualizing music artists"
   - Redirect URI: `http://127.0.0.1:8000/callback`
   - Website: `http://127.0.0.1:8000`
5. Accept the terms and create the app

## 2. Get Your Credentials

After creating the app, you'll see:
- Client ID
- Client Secret (click "Show Client Secret" to reveal it)

## 3. Set Environment Variables

Set these environment variables before running the application:

```bash
export SPOTIFY_CLIENT_ID="your_client_id_here"
export SPOTIFY_CLIENT_SECRET="your_client_secret_here"
```

Or create a `.env` file in the project root:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## 4. Features

- **OAuth Authentication**: Users connect their Spotify account
- **Cookie-based Sessions**: Tokens stored in cookies for persistence
- **Automatic Token Refresh**: Handles expired tokens automatically
- **Embedded Players**: Spotify tracks play directly in the app
- **Alternative Tracks**: Shows multiple tracks by the same artist
- **Autoplay Support**: Tracks can autoplay when clicked

## 5. Usage

1. Start the application
2. Click on any artist point
3. If not authenticated, click "Connect Spotify Account"
4. Authorize the application in Spotify
5. Return to the app and click the artist again
6. Music will play automatically in the embedded player

## 6. Security Notes

- Tokens are stored in HTTP-only cookies
- Access tokens expire after 1 hour
- Refresh tokens are used to get new access tokens
- No sensitive data is stored on the server
