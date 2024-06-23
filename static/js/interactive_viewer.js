function setup() {
    content = document.getElementById("content-ggn18");

    scenes = [
        ["Cutlery", "scenes/cutlery/"],
        ["Car", "scenes/car/"],
        ["Snail", "scenes/snail/"],
        ["Teaser", "scenes/teaser/"],
    ];
    new ImageBox(content, data['imageBoxes'], scenes, "ggn18")
}