const canvas = document.querySelector("canvas"),
toolBtns = document.querySelectorAll(".tool"),
sizeSlider = document.querySelector("#size-slider"),
clearCanvas = document.querySelector(".clear-canvas"),
saveImg = document.querySelector(".save-img"),
ctx = canvas.getContext("2d");

// global variables with default value
let prevMouseX, prevMouseY, snapshot,
isDrawing = false,
selectedTool = "brush",
brushWidth = 5,
selectedColor = "#000",
selectedColour = "#972626";

const canvasWidthInput = document.getElementById('canvas-width'),
      canvasHeightInput = document.getElementById('canvas-height'),
      resizeButton = document.getElementById('resize-canvas');

resizeButton.addEventListener('click', () => {
    const newWidth = parseInt(canvasWidthInput.value, 10);
    const newHeight = parseInt(canvasHeightInput.value, 10);
    
    // Change the canvas size
    canvas.width = newWidth;
    canvas.height = newHeight;
    canvas.style.width = newWidth + "px";
    canvas.style.height = newHeight + "px";

    // You might want to reset the canvas background here as well
    setCanvasBackground();
});

const setCanvasBackground = () => {
    // setting whole canvas background to white, so the downloaded img background will be white
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = selectedColor; // setting fillstyle back to the selectedColor, it'll be the brush color
}

window.addEventListener("load", () => {
    // setting canvas width/height.. offsetwidth/height returns viewable width/height of an element
    canvas.width = parseInt(500, 10);
    canvas.height = parseInt(600, 10);
    canvas.style.width = parseInt(500, 10) + "px";
    canvas.style.height = parseInt(600, 10) + "px";
    setCanvasBackground();
});

const drawRect = (e) => {
    if (e.type === "mousemove") {
        x = e.offsetX;
        y = e.offsetY;
        console.log("User clicked the point x", x, "y", y);
        ctx.fillStyle = selectedColour;
        ctx.fillFlood(x, y, 0);
    } 
}

const startDraw = (e) => {
    isDrawing = true;
    prevMouseX = e.offsetX; // passing current mouseX position as prevMouseX value
    prevMouseY = e.offsetY; // passing current mouseY position as prevMouseY value
    ctx.beginPath(); // creating new path to draw
    ctx.lineWidth = brushWidth; // passing brushSize as line width
    ctx.strokeStyle = selectedColor; // passing selectedColor as stroke style
    ctx.fillStyle = selectedColor; // passing selectedColor as fill style
    // copying canvas data & passing as snapshot value.. this avoids dragging the image
    snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
}

const drawing = (e) => {
    if(!isDrawing) return; // if isDrawing is false return from here
    ctx.putImageData(snapshot, 0, 0); // adding copied canvas data on to this canvas
    console.log(e.type)
    if(selectedTool === "brush" || selectedTool === "eraser") {
        // if selected tool is eraser then set strokeStyle to white 
        // to paint white color on to the existing canvas content else set the stroke color to selected color
        ctx.strokeStyle = selectedTool === "eraser" ? "#fff" : selectedColor;
        ctx.lineTo(e.offsetX, e.offsetY); // creating line according to the mouse pointer
        ctx.stroke(); // drawing/filling line with color
    } else if (selectedTool === "rectangle" && e.type === "mousemove") {
        console.log(e.type)
        drawRect(e);
    }
}

// Select different tools
toolBtns.forEach(btn => {
    btn.addEventListener("click", () => { // adding click event to all tool option
        // removing active class from the previous option and adding on current clicked option
        document.querySelector(".options .active").classList.remove("active");
        btn.classList.add("active");
        selectedTool = btn.id;
    });
});

// Change brush size
sizeSlider.addEventListener("change", () => brushWidth = sizeSlider.value); // passing slider value as brushSize

// Clear Canvas
clearCanvas.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // clearing whole canvas
    setCanvasBackground();
});

// Save Image
saveImg.addEventListener("click", () => {
    const link = document.createElement("a"); // creating <a> element
    link.download = `${Date.now()}.jpg`; // passing current date as link download value
    const canvasData = canvas.toDataURL();
    link.href = canvasData; // passing canvasData as link href value
    link.click(); // clicking link to download image

    // Send the image data to the backend
    fetch('/generate_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ imageData: canvasData })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server response: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        // Assuming the backend sends the processed image as a base64 encoded string
        const processedImageData = data.processedImage;
    
        // Trigger download
        const link = document.createElement("a");
        link.href = processedImageData;
        link.download = `${Date.now()}.jpg`; // or .jpg or any other format depending on the data you get back
        document.body.appendChild(link); // temporarily add link to document
        link.click();
        document.body.removeChild(link); // remove the temporary link
    })
    .catch(error => {
        console.error('Error receiving the processed image from the server:', error.message);
    });
});

// Upload image
document.getElementById('uploadButton').addEventListener('click', () => {
    // Trigger file input click to open file dialog
    document.getElementById('uploadImg').click();
});

document.getElementById('uploadImg').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    // Convert the image to a Data URL
    const reader = new FileReader();
    reader.onloadend = () => {
        // This is the data we will send to the server
        const imageData = reader.result;

        // Send the image data to the backend
        fetch('/generate_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imageData: imageData })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server response: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Assuming the backend sends the processed image as a base64 encoded string
            const processedImageData = data.processedImage;
        
            // Trigger download of the processed image
            const link = document.createElement("a");
            link.href = processedImageData;
            link.download = `${Date.now()}.jpg`; // or .png, .jpeg, etc.
            document.body.appendChild(link); // temporarily add link to document
            link.click();
            document.body.removeChild(link); // remove the temporary link
        })
        .catch(error => {
            console.error('Error receiving the processed image from the server:', error.message);
        });
    };
    reader.readAsDataURL(file);
});

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", drawing);
canvas.addEventListener("mouseup", () => isDrawing = false);
canvas.addEventListener("mouseout", () => isDrawing = false);
canvas.addEventListener("mouseleave", () => isDrawing = false);

document.getElementById("colorForm").addEventListener("change", (evt) => {
    selectedColour = evt.target.value;
    const activeOption = document.querySelector(".options .active");
    if (activeOption) {
        activeOption.classList.remove("active");
    }
    const btn = document.getElementById("rectangle");
    btn.classList.add("active");
    selectedTool = btn.id;
});