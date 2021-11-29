IMAGE=cv2
run:
	docker build -t ${IMAGE} .
in:
	nvidia-docker run --runtime=nvidia -v /home/hagi:/opt --rm -it ${IMAGE}:latest
