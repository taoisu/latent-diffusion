#!/bin/bash

[ "$TAG" ]

DOCKER_BUILDKIT=1 sudo docker build --pull --rm \
  --secret id=npm,src=.npmrc \
  -f Dockerfile -t oneocracr.azurecr.io/verticals/painter:$TAG ..

az acr login --name oneocracr
sudo docker push oneocracr.azurecr.io/verticals/painter:$TAG