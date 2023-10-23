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
    if (selectedTool === "rectangle") {
        console.log(selectedTool)
        addFormListener();
        canvas.addEventListener("click", (evt) => {
            const { x, y } = getEventCoords(evt, canvas.getBoundingClientRect());
            console.log("User clicked the point x", x, "y", y);
            ctx.fillStyle = selectedColour;
            ctx.fillFlood(x, y, 0);
        });
    } else {
        return
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

    if (selectedTool === "rectangle") {
        addFormListener();
        canvas.addEventListener("click", (evt) => {
            const { x, y } = getEventCoords(evt, canvas.getBoundingClientRect());
            console.log("User clicked the point x", x, "y", y);
            ctx.fillStyle = selectedColour;
            ctx.fillFlood(x, y, 0);
        });
    } else if(selectedTool === "brush" || selectedTool === "eraser") {
        // if selected tool is eraser then set strokeStyle to white 
        // to paint white color on to the existing canvas content else set the stroke color to selected color
        ctx.strokeStyle = selectedTool === "eraser" ? "#fff" : selectedColor;
        ctx.lineTo(e.offsetX, e.offsetY); // creating line according to the mouse pointer
        ctx.strokeStyle = selectedColor;
        ctx.stroke(); // drawing/filling line with color
    }
}

// Select different tools
toolBtns.forEach(btn => {
    btn.addEventListener("click", () => { // adding click event to all tool option
        // removing active class from the previous option and adding on current clicked option
        console.log(document.querySelector(".options .active"))
        document.querySelector(".options .active").classList.remove("active");
        btn.classList.add("active");
        selectedTool = btn.id;
        console.log(selectedTool);
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
    link.href = canvas.toDataURL(); // passing canvasData as link href value
    link.click(); // clicking link to download image
});

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", drawing);
canvas.addEventListener("mouseup", () => isDrawing = false);

function addFormListener() {
    document.getElementById("colorForm").addEventListener("change", (evt) => {
        selectedColour = evt.target.value;

        // // Remove "active" class from the currently active option
        // const activeOption = document.querySelector(".options .active");
        // if (activeOption) {
        //     activeOption.classList.remove("active");
        // }

        // // Add "active" class to the rectangle tool and set selectedTool to "rectangle"
        // const btn = document.getElementById("rectangle");
        // btn.classList.add("active");
        // selectedTool = btn.id;
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