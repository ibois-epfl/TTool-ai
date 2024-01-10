# TTool-AI

TTool-AI automates the integration of new tools into AC (Augmented Carpentry), enhancing efficiency and simplifying the process.

## System dependencies

1. **Install Docker and Docker Compose**:

    Ensure you have Docker and Docker Compose installed on your system with NVIDIA Runtime support for the Training Service.

2. **Environment Variables**:
    TTool-AI relies on environment variables defined in a .env file. 
    Make sure to set up this file as per the project's requirements.


## Getting Started

### Clone the repository:

```bash
git clone git@github.com:ibois-epfl/TTool-ai.git
```

### Run the project:

Navigate to the project's root directory and run the following command:
```bash
cd TTool-ai/
```
Run Docker Compose to build the project in the background:
```bash
docker-compose up -d
```
Run Docker Compose to build the project in the foreground:
```bash
docker-compose up
```
   
### Check the status of the containers:

```bash
docker-compose ps -a
```
### Access the Service:
Once everything is up and running, you can access the FastAPI interface at:
```bash
http://localhost:16666/docs
```


