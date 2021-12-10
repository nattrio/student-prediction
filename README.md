# fasiapi-cd

[Tutorial](https://towardsdatascience.com/deploy-fastapi-on-azure-with-github-actions-32c5ab248ce3)

## How to run with Docker

```bash
# Build Docker Image
docker build -t fastapi-cd:1.0 .

# Run API service on port 8000
docker run -p 8000:8000 --name fastapi fastapi-cd:1.0
```
