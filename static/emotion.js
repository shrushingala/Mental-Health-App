function startCamera() {
    const videoElement = document.createElement("video");
    videoElement.setAttribute("id", "webcam");
    document.body.appendChild(videoElement);

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoElement.srcObject = stream;
            videoElement.play();
        })
        .catch(error => {
            console.error("Error accessing webcam:", error);
        });
}
