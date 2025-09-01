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
        { title: "How to Hide the Emperor'...", image: "how to hide the emperor's child.png" },
        { title: "Morgana and Oz", image: "Morgana-and-Oz.png" },
        { title: "The Reborn Young Lord is ...", image: "The_Reborn_Young_Lord_is_an_Assassin.png" },
        { title: "The Extra's Academy Survi...", image: "The Extra’s Academy Survival Guide.png"},
        { title: "The Reason for the twin L...", image: "the reason for the twin lady's disguise.jpg"},



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
        { title: "Behind Her Highness’s Smile", image: "behind-her-highnesss-smile.png"}, 
    ],
    THU: [
        { title: "Tears on a Withered Flower", image: "Tears on a Withered Flower.png" },
        { title: "Vampire family", image: "vampireFamily.png" },
        { title: "To Whom It No Longer Co...", image: "To Whom It No Longer Concerns.jpg" },
        { title: "The Greatest Estate Devel...", image: "The_Greatest_Estate_Developer_poster.png" }


    ],
    FRI: [
        { title: "The Knight Only Lives Today", image: "The Knight Only Lives Today.png"},
        { title: "Operation True Love", image: "Operation True Love.png" }
    ],
    SAT: [
        { title: "The Regressed Empress", image: "The Regressed Empress's abduction marriage.png" },
        { title: "Hidden Secrets", image: "sat2.jpg" }
    ],
    SUN: [
        // { title: "The Regressed Empresshe Regressed Empresss", image: "The Regressed Empress's abduction marriage.png" },
        { title: "This wasn't in my adoptio...", image: "this wasn't in my adoption plan.png" }
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
function generateList() {
  const genre = document.getElementById("genre").value;
  const mood = document.getElementById("mood").value;
  const list = document.getElementById("list");

  // Clear old results
  list.innerHTML = "";

  // Sample AI picks (you can expand this)
  const recommendations = {
    fantasy: ["Tower of God", "Lore Olympus", "Omniscient Reader"],
    romance: ["Let's Play", "Age Matters", "Siren's Lament"],
    comedy: ["The God of High School", "Blue Chair", "My Dictator Boyfriend"],
    drama: ["I Love Yoo", "Unordinary", "Sweet Home"],
    "sci-fi": ["Nano List", "Eleceed", "Flow"]
  };

  const moodBoost = {
    happy: "💖 Fun & uplifting vibes!",
    adventurous: "⚔️ Full of action and thrill!",
    emotional: "😭 Deep & heartfelt stories!",
    chill: "😌 Relaxing reads to wind down!"
  };

  // AI "recommendation"
  const picks = recommendations[genre];
  const chosen = picks.sort(() => 0.5 - Math.random()).slice(0, 3); // random 3

  // Show output
  chosen.forEach(item => {
    const li = document.createElement("li");
    li.textContent = `${item} — ${moodBoost[mood]}`;
    list.appendChild(li);
  });
}
