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
        { title: "Trapped in a Soap Opera", image: "School Bus Graveyard.png" },
        { title: "I was the final boss", image: "behind-her-highnesss-smile.png" },
        { title: "Re: Trailer Trash", image: "re trailer trash.png" },
        
        

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
    ]
};

// Render series cards for today
const container = document.querySelector(".series-container");
container.innerHTML = ""; // clear previous cards

const todayKey = days[todayIndex];
if(seriesData[todayKey]){
    seriesData[todayKey].forEach(series => {
        const card = document.createElement("div");
        card.className = "series-card";
        card.innerHTML = `
            <img src="${series.image}" alt="${series.title}">
            <h3>${series.title}</h3>
        `;
        container.appendChild(card);
    });
}
