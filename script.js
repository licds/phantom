const canvas = document.getElementById("sketch-canvas");
const ctx = canvas.getContext("2d");
const colorPicker = document.getElementById("color-picker");
const brushSize = document.getElementById("brush-size");
const clearButton = document.getElementById("clear-canvas");
let isDrawing = false;
let isFilling = false;
let fillColor = "#000000"; // Default fill color

canvas.addEventListener("mousedown", () => {
    isDrawing = true;
    isFilling = false; // Ensure filling is off while drawing
    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = brushSize.value;
});

canvas.addEventListener("mousemove", (e) => {
    if (isDrawing) {
        ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        ctx.stroke();
    }
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
    ctx.closePath();
});

// Enable fill functionality when the canvas is clicked
canvas.addEventListener("click", (e) => {
    isFilling = true;
    fillColor = colorPicker.value;
    fill(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
});

colorPicker.addEventListener("input", () => {
    ctx.strokeStyle = colorPicker.value;
    if (!isFilling) {
        fillColor = colorPicker.value; // Update fill color when drawing
    }
});

clearButton.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Fill function
function fill(x, y) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixelStack = [{ x, y }];
    const targetColor = getPixelColor(x, y);

    if (targetColor === fillColor) {
        return; // If the selected area already has the target color, no need to fill.
    }

    while (pixelStack.length) {
        const currentPosition = pixelStack.pop();
        const { x, y } = currentPosition;

        if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
            const currentColor = getPixelColor(x, y);

            if (currentColor === targetColor) {
                setPixelColor(x, y, fillColor);
                pixelStack.push({ x: x - 1, y });
                pixelStack.push({ x: x + 1, y });
                pixelStack.push({ x, y: y - 1 });
                pixelStack.push({ x, y: y + 1 });
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

function getPixelColor(x, y) {
    const pixelIndex = (y * canvas.width + x) * 4;
    const r = imageData.data[pixelIndex];
    const g = imageData.data[pixelIndex + 1];
    const b = imageData.data[pixelIndex + 2];
    return `rgb(${r},${g},${b})`;
}

function setPixelColor(x, y, color) {
    const pixelIndex = (y * canvas.width + x) * 4;
    const [r, g, b] = color.match(/\d+/g);
    imageData.data[pixelIndex] = r;
    imageData.data[pixelIndex + 1] = g;
    imageData.data[pixelIndex + 2] = b;
}

let selectedFillColor = "#ff0000"; // Default fill color

// Add event listeners to the radio buttons for fill color selection
const fillColorRadioButtons = document.querySelectorAll('input[name="fill-color"]');
fillColorRadioButtons.forEach((radio) => {
    radio.addEventListener("change", () => {
        selectedFillColor = radio.value;
    });
});

// Add the fill bucket tool (using only solid fill)
canvas.addEventListener("click", (e) => {
    fillSolid(selectedFillColor, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
});

// Define the fillSolid function
function fillSolid(targetColor, startX, startY) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixelStack = [{ x: startX, y: startY }];

    const targetColorRGBA = hexToRGBA(targetColor);
    const startIndex = (startY * canvas.width + startX) * 4;

    const startColor = {
        r: imageData.data[startIndex],
        g: imageData.data[startIndex + 1],
        b: imageData.data[startIndex + 2],
        a: imageData.data[startIndex + 3],
    };

    if (colorMatch(startColor, targetColorRGBA)) {
        return; // If the start color already matches the target color, no need to fill.
    }

    while (pixelStack.length) {
        const currentPosition = pixelStack.pop();
        const { x, y } = currentPosition;

        const currentPosIndex = (y * canvas.width + x) * 4;
        while (y >= 0 && colorMatch(startColor, targetColorRGBA, currentPosIndex)) {
            y--;
            currentPosIndex -= canvas.width * 4;
        }

        currentPosIndex += canvas.width * 4;
        y++;
        let reachLeft = false;
        let reachRight = false;

        while (y < canvas.height - 1 && colorMatch(startColor, targetColorRGBA, currentPosIndex)) {
            y++;

            fillPixel(currentPosIndex);

            if (x > 0) {
                if (colorMatch(startColor, targetColorRGBA, currentPosIndex - 4)) {
                    if (!reachLeft) {
                        pixelStack.push({ x: x - 1, y });
                        reachLeft = true;
                    }
                } else if (reachLeft) {
                    reachLeft = false;
                }
            }

            if (x < canvas.width - 1) {
                if (colorMatch(startColor, targetColorRGBA, currentPosIndex + 4)) {
                    if (!reachRight) {
                        pixelStack.push({ x: x + 1, y });
                        reachRight = true;
                    }
                } else if (reachRight) {
                    reachRight = false;
                }
            }

            currentPosIndex += canvas.width * 4;
        }
    }

    ctx.putImageData(imageData, 0, 0);

    function fillPixel(index) {
        imageData.data[index] = targetColorRGBA.r;
        imageData.data[index + 1] = targetColorRGBA.g;
        imageData.data[index + 2] = targetColorRGBA.b;
        imageData.data[index + 3] = targetColorRGBA.a;
    }

    function colorMatch(color1, color2, index) {
        index = index || 0;
        const r1 = imageData.data[index];
        const g1 = imageData.data[index + 1];
        const b1 = imageData.data[index + 2];
        const a1 = imageData.data[index + 3];

        return (
            r1 === color1.r &&
            g1 === color1.g &&
            b1 === color1.b &&
            a1 === color1.a
        );
    }

    function hexToRGBA(hex) {
        hex = hex.replace(/^#/, '');
        const bigint = parseInt(hex, 16);
        return {
            r: (bigint >> 16) & 255,
            g: (bigint >> 8) & 255,
            b: bigint & 255,
            a: 255,
        };
    }
}

// Add size buttons and their event listeners
const sizeButtons = document.querySelectorAll(".size-buttons button");
sizeButtons.forEach(button => {
    button.addEventListener("click", () => {
        sizeButtons.forEach(btn => btn.classList.remove("active"));
        button.classList.add("active");
        setCanvasSize(button.id);
    });
});

function setCanvasSize(size) {
    switch (size) {
        case "size-small":
            canvas.width = 200;
            canvas.height = 400;
            break;
        case "size-medium":
            canvas.width = 300;
            canvas.height = 400;
            break;
        case "size-large":
            canvas.width = 500;
            canvas.height = 500;
            break;
        default:
            break;
    }
}

canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = brushSize.value;
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
});

canvas.addEventListener("mousemove", (e) => {
    if (!isDrawing) return;
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
    ctx.closePath();
});

colorPicker.addEventListener("input", () => {
    ctx.strokeStyle = colorPicker.value;
});

brushSize.addEventListener("input", () => {
    ctx.lineWidth = brushSize.value;
});

clearButton.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});
