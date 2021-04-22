# DETOXIS (IberLEF 2021)

The aim of the DETOXIS (DEtection of TOxicity in comments In Spanish) task is the detection of toxicity in comments posted in Spanish in response to different online news articles related to immigration. This task is divided into two related classification subtasks: Toxicity detection task and Toxicity level detection task.

The presence of toxic messages on social media and the need to identify and mitigate them leads to the development of systems for their automatic detection. The automatic detection of toxic language, especially in tweets and comments, is a task that has attracted growing interest from the NLP community in recent years. This interest is reflected in the diversity of the shared tasks that have been organized recently, among which we highlight those held over the last two years: HateEval-2019[1] (Basile et al., 2019) on hate speech against immigrants and women in English and Spanish tweets; TRAC-2 task on Aggression Identification[2] (Kumar et al., 2020) for English, Bengali and Hindi in comments extracted from YouTube; the OffensEval-2020[3] on offensive language identification (Zampieri et al., 2020) in Arabic, Danish, English, Greek and Turkish tweets; GermEval-2019 shared task on the Identification of Offensive Language for German[4] on Twitter (Struẞ et al. 2019); and the Jigsaw Multilingual Toxic Comment Classification Challenge[5], in which the task is focused on building multilingual models (English, French, German, Italian, Portuguese, Russian and Spanish) with English-only training data from Wikipedia comments.

DETOXIS is the first task that focuses on the detection of different levels of toxicity in comments posted in response to news articles written in Spanish. For more information regarding this task, click <a href="https://detoxisiberlef.wixsite.com/website">here</a>.

This repository contains the necessary code to:

<ol>
  <li>Read and write results in the correct submission format</li>
  <li>Evaluate your models with all considered evaluation metrics using the Evaluator class</li>
</ol>

Furthermore, several baselines have been included for comparison purposes as well as a cross-validation example using these baselines.
