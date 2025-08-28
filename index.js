// Days array for buttons
const days = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
const todayIndex = new Date().getDay(); // 0-6

// Highlight current day button
const buttons = document.querySelectorAll(".special-btn");
buttons.forEach(btn => btn.classList.remove("active"));
buttons.forEach(btn => {
    if (btn.textContent === days[todayIndex]) {
        btn.classList.add("active");
    }
});

// Example series list for each day
const seriesData = {
    MON: [
        { title: "Mystery High", image: "trapped in a soap opera.png" },
        { title: "Tech Warriors", image: "mon2.jpg" }
    ],
    TUE: [
        { title: "Love in Tokyo", image: "tue1.jpg" },
        { title: "Cyber Ninjas", image: "tue2.jpg" }
    ],
    WED: [
        { title: "Trapped in a soap Opera", image: "trapped in a soap opera.png" },
        { title: "I was the final boss", image: "I_was_the_final_boss.png" },
        { title: "Re: Trailer Trash", image: "re trailer trash.png" },
        { title: "School Bus Graveyard", image: "School Bus Graveyard.png" },
        { title: "Behind Her Highness’s Smile", image: "behind-her-highnesss-smile.png" }
    ],
    THU: [
        { title: "Robot Rebellion", image: "thu1.jpg" },
        { title: "Ocean Adventures", image: "thu2.jpg" }
    ],
    FRI: [
        { title: "Space Explorers", image: "fri1.jpg" },
        { title: "Detective Stories", image: "fri2.jpg" }
    ],
    SAT: [
        { title: "Magic Academy", image: "sat1.jpg" },
        { title: "Hidden Secrets", image: "sat2.jpg" }
    ],
    SUN: [
        { title: "Legendary Heroes", image: "sun1.jpg" },
        { title: "Romantic Saga", image: "sun2.jpg" }
    ],
    COMPLETED:[
        { title: "D Heroes", image: "sun1.jpg" },
        { title: "Romantic Saga", image: "sun2.jpg" }    
    ]
};

const container = document.querySelector(".series-container");

// Function to render series for a given day
function renderSeries(dayKey) {
    container.innerHTML = ""; // clear previous cards
    if (seriesData[dayKey]) {
        seriesData[dayKey].forEach(series => {
            const card = document.createElement("div");
            card.className = "series-card";
            card.innerHTML = `
                <img src="${series.image}" alt="${series.title}">
                <h3>${series.title}</h3>
            `;
            container.appendChild(card);
        });
    }
}

// Initial render for today
renderSeries(days[todayIndex]);

// Handle button clicks
buttons.forEach(btn => {
    btn.addEventListener("click", () => {
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const dayKey = btn.textContent; // Button text is "MON", "TUE", etc.
        renderSeries(dayKey);
    });
});
