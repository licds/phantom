const canvas = document.querySelector("canvas"),
toolBtns = document.querySelectorAll(".tool"),
sizeSlider = document.querySelector("#size-slider"),
colorBtns = document.querySelectorAll(".colors .option"),
colorPicker = document.querySelector("#color-picker"),
clearCanvas = document.querySelector(".clear-canvas"),
saveImg = document.querySelector(".save-img"),
ctx = canvas.getContext("2d");

// global variables with default value
let prevMouseX, prevMouseY, snapshot,
isDrawing = false,
selectedTool = "brush",
brushWidth = 5,
selectedColor = "#000",
selectedColour = "#00FF00";

const setCanvasBackground = () => {
    // setting whole canvas background to white, so the downloaded img background will be white
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = selectedColor; // setting fillstyle back to the selectedColor, it'll be the brush color
}

window.addEventListener("load", () => {
    // setting canvas width/height.. offsetwidth/height returns viewable width/height of an element
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    setCanvasBackground();
});

const drawRect = (e) => {
    addFormListener();
    canvas.addEventListener("click", (evt) => {
        const { x, y } = getEventCoords(evt, canvas.getBoundingClientRect());
        console.log("User clicked the point x", x, "y", y);
        // fillColour(x, y, canvas, ctx, selectedColour);
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const floodFill = new FloodFill(imgData);
        floodFill.fill(selectedColour, x, y, 0);
        ctx.putImageData(floodFill.imageData, 0, 0)
    });
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

    if(selectedTool === "brush" || selectedTool === "eraser") {
        // if selected tool is eraser then set strokeStyle to white 
        // to paint white color on to the existing canvas content else set the stroke color to selected color
        ctx.strokeStyle = selectedTool === "eraser" ? "#fff" : selectedColor;
        ctx.lineTo(e.offsetX, e.offsetY); // creating line according to the mouse pointer
        ctx.stroke(); // drawing/filling line with color
    } else if(selectedTool === "rectangle"){
        drawRect(e);
    }
}

toolBtns.forEach(btn => {
    btn.addEventListener("click", () => { // adding click event to all tool option
        // removing active class from the previous option and adding on current clicked option
        document.querySelector(".options .active").classList.remove("active");
        btn.classList.add("active");
        selectedTool = btn.id;
    });
});

sizeSlider.addEventListener("change", () => brushWidth = sizeSlider.value); // passing slider value as brushSize

colorBtns.forEach(btn => {
    btn.addEventListener("click", () => { // adding click event to all color button
        // removing selected class from the previous option and adding on current clicked option
        document.querySelector(".options .selected").classList.remove("selected");
        btn.classList.add("selected");
        // passing selected btn background color as selectedColor value
        selectedColor = window.getComputedStyle(btn).getPropertyValue("background-color");
    });
});

colorPicker.addEventListener("change", () => {
    // passing picked color value from color picker to last color btn background
    colorPicker.parentElement.style.background = colorPicker.value;
    colorPicker.parentElement.click();
});

clearCanvas.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // clearing whole canvas
    setCanvasBackground();
});

saveImg.addEventListener("click", () => {
    const link = document.createElement("a"); // creating <a> element
    link.download = `${Date.now()}.jpg`; // passing current date as link download value
    link.href = canvas.toDataURL(); // passing canvasData as link href value
    link.click(); // clicking link to download image
});

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", drawing);
canvas.addEventListener("mouseup", () => isDrawing = false);

function fillColour(x, y, canvas, ctx, fillColor) {
    const width = canvas.width;
    const height = canvas.height;
    const canvasData = ctx.getImageData(0, 0, width, height);
    const originalColor = localColor(x, y);
    fillColor = hexToRgb(fillColor)

    function getColorIndex(x, y) {
        return (y * width + x) * 4;
    }
    function localColor(index) {
        const i = getColorIndex(x, y);
        const r = canvasData.data[i];
        const g = canvasData.data[i + 1];
        const b = canvasData.data[i + 2];
        console.log("localColor", `rgb(${r},${g},${b})`);
        return `rgb(${r},${g},${b})`;
    }
    function fillColorAt(x, y) {
        console.log("fillcolor", fillColor);
        const i = getColorIndex(x, y);
        canvasData.data[i] = fillColor[0];
        canvasData.data[i + 1] = fillColor[1];
        canvasData.data[i + 2] = fillColor[2];
        // canvasData.data[index + 3] = 255; // Set the alpha channel to fully opaque
    }
    function hexToRgb(hex) {
        hex = hex.replace(/^#/, '');
        const bigint = parseInt(hex, 16);
        const r = (bigint >> 16) & 255;
        const g = (bigint >> 8) & 255;
        const b = bigint & 255;
        return { r, g, b };
    }
    function floodFillRecursive(x, y) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            console.log("x", x, "y", y);
            return;
        }
        if (localColor(x, y) !== originalColor) {
            console.log("localColor != originalColor", localColor(x, y));
            return;
        }
    
        fillColorAt(x, y);
    
        floodFillRecursive(x + 1, y);
        floodFillRecursive(x - 1, y);
        floodFillRecursive(x, y + 1);
        floodFillRecursive(x, y - 1);
    }
    

    if (localColor(x, y) !== fillColor) {
        floodFillRecursive(x, y);
        ctx.putImageData(canvasData, 0, 0);
    }
}

function addFormListener() {
    document.getElementById("colorForm").addEventListener("change", (evt) => {
      selectedColour = evt.target.value;
    });
}

function getEventCoords(evt, nodeRect) {
    let x, y;
    if (evt.touches && evt.touches.length > 0) {
      x = evt.touches[0].clientX;
      y = evt.touches[0].clientY;
    } else {
      x = evt.clientX;
      y = evt.clientY;
    }
    return { x: Math.round(x - nodeRect.x), y: Math.round(y - nodeRect.y) };
  }