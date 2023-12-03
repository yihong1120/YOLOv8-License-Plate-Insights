🇬🇧 [English](./Flask_GAE_Deployment_Guide.md) | 🇹🇼 [繁體中文](./Flask_GAE_Deployment_Guide-zh.tw.md)

# Deploying a Flask API to Google App Engine

This guide provides a step-by-step process for deploying a Flask-based API to Google App Engine (GAE), allowing you to host and manage your application efficiently.

## Prerequisites

- A Flask API developed in your local environment.
- Python 3.6 or higher installed.
- Google Cloud SDK installed on your machine.
- A Google Cloud account.

## Step 1: Prepare Your Flask Application

Ensure your Flask application is running properly in your local environment. The application should be in a complete state for deployment.

## Step 2: Create the `app.yaml` Configuration File

In the root directory of your Flask application, create a file named `app.yaml`. This file contains configuration settings for GAE. For a basic Flask application, the contents should be:

```yaml
runtime: python39
entrypoint: gunicorn -b :$PORT main:app
```

Replace `main:app` with the appropriate module and application name of your Flask app.

## Step 3: Configure `requirements.txt`

Make sure your `requirements.txt` file lists all the necessary dependencies. It should at least include Flask and Gunicorn:

```
Flask==X.X.X
gunicorn==X.X.X
# Other dependencies...
```

## Step 4: Install Google Cloud SDK

If not already installed, download and install the Google Cloud SDK. This tool is crucial for deploying and managing your application on GAE.

## Step 5: Initialize Your GAE Project

Run the following command to initialise your GAE configuration:

```bash
gcloud init
```

Follow the prompts to select or create a new Google Cloud project and set a default region.

## Step 6: Deploy Your Application

Navigate to your Flask app's directory and run:

```bash
gcloud app deploy
```

This command uploads and deploys your application to GAE. Upon successful deployment, you will receive a URL to access your app.

## Step 7: Access Your Application

Open the provided URL in a browser. Your Flask application should now be running online.

## Additional Notes

- Ensure that the App Engine service is enabled in your Google Cloud project.
- Regularly monitor your application to ensure it runs as expected.
- Consider the security and resource usage of your application, especially if expecting high traffic.

Deploying to GAE allows you to focus on development while Google handles infrastructure management tasks such as server scaling and maintenance.
This guide covers the essential steps for deploying a Flask API to Google App Engine, intended for developers with a basic understanding of Flask and Google Cloud services.