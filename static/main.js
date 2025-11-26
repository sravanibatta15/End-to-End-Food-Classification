// Load nutrition when class changes
document.getElementById("class-dropdown").addEventListener("change", async function () {
    const selectedClass = this.value;
    const box = document.getElementById("nutritionBox");

    if (!selectedClass) {
        box.innerHTML = "<h3>Nutrition Details</h3><p>Select class</p>";
        return;
    }

    const res = await fetch("/get_nutrition?class=" + selectedClass);
    const data = await res.json();

    box.innerHTML = `<h3>Nutrition Details</h3>`;
    for (let key in data) {
        box.innerHTML += `<p><b>${key}:</b> ${data[key]}</p>`;
    }
});
