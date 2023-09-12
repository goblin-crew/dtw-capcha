// Select the image element using the query selector
const imageElement = document.querySelector("#content > div > div.eight.columns > div:nth-child(1) > div.block-content.row.level > div > div:nth-child(1) > div > p:nth-child(1) > img");

const input = document.querySelector("#answer")
const submitButton = document.querySelector("#content > div > div.eight.columns > div:nth-child(1) > div.block-content.row.level > div > div:nth-child(2) > div > div:nth-child(1) > div > form > button")

// Check if the image element exists
if (imageElement) {
    // Create a canvas element
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    // Set the canvas dimensions to match the image
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;

    // Draw the image onto the canvas
    context.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

    // Get the binary data of the image from the canvas as a Data URL
    const imageDataUrl = canvas.toDataURL("image/png");

    // Define the server URL
    const serverUrl = "http://127.0.0.1:5000"; // Change this to your server's URL

    // Create a FormData object to send the image data as a file
    const formData = new FormData();
    formData.append("image", dataURItoBlob(imageDataUrl));

    // Make a POST request to the server
    fetch(serverUrl, {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((responseJson) => {
            // Handle the response from the server
            console.log("Server Response:", responseJson);
            input.value = responseJson.split("").reverse().join("")
            submitButton.click()
        })
        .catch((error) => {
            console.error("Error uploading image:", error);
        });
} else {
    console.error("Image element not found");
}

// Helper function to convert Data URL to Blob
function dataURItoBlob(dataURI) {
    const byteString = atob(dataURI.split(",")[1]);
    const mimeString = dataURI.split(",")[0].split(":")[1].split(";")[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
}
