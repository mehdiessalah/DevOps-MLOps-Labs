FROM ubuntu:latest
LABEL authors="mehdi"

ENTRYPOINT ["top", "-b"]