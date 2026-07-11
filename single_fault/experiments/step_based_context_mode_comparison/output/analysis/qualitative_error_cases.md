# Qualitative Error Cases

## Prev Beats Next

| file | gt_agent | gt_step | trajectory_length | prev_pred_agent | prev_pred_step | next_pred_agent | next_pred_step | question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19.json | WebSurfer | 12 | 29 | WebSurfer | 12 | Orchestrator (thought) | 9 | Where can I take martial arts classes within a five-minute walk from the New York Stock Exchange after work (7-9 pm)? |
| 29.json | Orchestrator | 1 | 5 | Orchestrator (thought) | 1 | Orchestrator (thought) | 2 | In the fictional language of Tizin, basic sentences are arranged with the Verb first, followed by the direct object, followed by the subject of the sentence. I want to express my love for apples to my Tizin friend. |
| 30.json | WebSurfer | 12 | 16 | Assistant | 12 | Orchestrator (thought) | 7 | In Unlambda, what exact charcter or text needs to be added to correct the following code to output "For penguins"? If what is needed is a character, answer with the name of the character. If there are different names for the character, use the shortest. The text location is not needed. Code: |
| 38.json | WebSurfer | 32 | 93 | WebSurfer | 32 | Orchestrator (thought) | 27 | During the first week of August 2015, one of the NASA Astronomy Pictures of the Day shows the lights of a city on the horizon. The namesake of this city also has a landmark building in Chicago named after him. What is the name of the architectural firm that designed this landmark building? Give the first name appearing in the name of the firm as of June 2023. |
| 43.json | Orchestrator | 3 | 52 | Orchestrator (-> WebSurfer) | 3 | WebSurfer | 20 | What are hikes in Yellowstone that have been recommended by at least three different people with kids and are highly rated on TripAdvisor (an average from 4.5/5 from at least 50 reviews)? |

## Next Beats Prev

| file | gt_agent | gt_step | trajectory_length | prev_pred_agent | prev_pred_step | next_pred_agent | next_pred_step | question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37.json | WebSurfer | 12 | 13 | Not Found | -1 | WebSurfer | 12 | According to the USGS, in what year was the American Alligator first found west of Texas (not including Texas)? |
| 55.json | WebSurfer | 8 | 74 | Orchestrator (thought) | 24 | WebSurfer | 8 | What is the cheapest option to mail a DVD to Colombia from Hartford, Connecticut using FedEx, DHL, or USPS? (The answer should be a json object with the keys "sender" and "price (usd)") |

## Both Wrong

| file | gt_agent | gt_step | trajectory_length | prev_pred_agent | prev_pred_step | next_pred_agent | next_pred_step | question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.json | WebSurfer | 3 | 67 | FileSurfer | 32 | FileSurfer | 28 | I read a paper about multiwavelength observations of fast radio bursts back in March 2021 on Arxiv, and it had a fascinating diagram of an X-ray time profile. There was a similar burst-1 diagram in another paper from one of the same authors about fast radio bursts back in July 2020, but I can't recall what the difference in seconds in the measured time span was. How many more seconds did one measure than the other? Just give the number. |
| 1.json | Orchestrator | 9 | 91 | Orchestrator (-> WebSurfer) | 10 | WebSurfer | 4 | What is the highest rated (according to IMDB) Daniel Craig movie that is less than 150 minutes and is available on Netflix (US)? |
| 10.json | WebSurfer | 4 | 59 | Orchestrator (thought) | 8 | Orchestrator (thought) | 19 | What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website? Just give the number. |
| 11.json | WebSurfer | 4 | 25 | WebSurfer | 24 | WebSurfer | 16 | On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by? |
| 12.json | Orchestrator | 51 | 67 | FileSurfer | 16 | FileSurfer | 16 | According to the World Bank, which countries had gross savings of over 35% of GDP for every year in the period 2001-2010? Give your answer as a comma-separated list of countries in alphabetical order. Use the countries most common names in english when answering. |
