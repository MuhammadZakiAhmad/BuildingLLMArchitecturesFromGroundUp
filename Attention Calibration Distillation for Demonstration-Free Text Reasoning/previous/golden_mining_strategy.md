# Phase 3 Mining Strategy: Maximizing "Golden Records" Yield

To successfully train the adapter in Phase 3, we need to extract thousands of "Golden Records" where the 0-shot Student fails, but the Few-shot Teacher is **highly confident and correct**. 

Currently, our mining script simply grabs the first 2 examples from the dataset and uses them as static 2-shot prompts for every single query. Based on current state-of-the-art research into In-Context Learning (ICL) optimization, this approach is highly inefficient and leaves massive performance on the table.

To drastically increase the Teacher's win rate (and thus harvest exponentially more golden records), we need to implement the following five strategies into our next mining script:

## 1. Scale Up to 4-Shot or 8-Shot
*   **The Science:** ICL performance scales logarithmically with the number of demonstrations ($k$). A 2-shot prompt barely establishes the pattern, whereas a 4-shot or 8-shot prompt fundamentally aligns the model's internal representations to the task manifold.
*   **Action:** We will increase the prompt from 2 examples to 4 examples. This will significantly boost the Teacher's accuracy and confidence scores, crossing the `> 0.60` threshold much more frequently.

## 2. Dynamic Similarity-Based Retrieval (k-NN)
*   **The Science:** Currently, if we ask a question about basketball, the static 2-shot prompt might give examples about hockey and golf. Research proves that LLMs perform substantially better when the demonstrations are semantically similar to the query.
*   **Action:** Instead of hardcoding the first 4 records of the dataset, we will use a lightweight embedding model (or simple TF-IDF) to find the 4 examples in the training set that are *most similar* to the current query, creating a dynamic, custom-tailored prompt for every single question.

## 3. Strict Diversity & Label Balancing
*   **The Science:** LLMs suffer from severe "Majority Label Bias". If a 4-shot prompt accidentally contains 3 examples where the answer is "A" and 1 where the answer is "B", the model will heavily lean toward guessing "A".
*   **Action:** When selecting the 4 examples, we will algorithmically enforce a perfect balance (e.g., two "A" examples and two "B" examples). This forces the Teacher to actually *reason* rather than just copying the most frequent label.

## 4. Mitigating Recency Bias (Order Randomization)
*   **The Science:** "Recency Bias" is a well-documented phenomenon where an LLM disproportionately weights the very last example right before the query. If the last example always has the answer "B", the model will artificially boost the probability of "B".
*   **Action:** Once the 4 examples are selected, we must dynamically shuffle their order for every single query to ensure the Teacher isn't learning a positional shortcut.

## 5. The Ultimate Weapon: Chain-of-Thought (CoT) Injection
*   **The Science:** Standard ICL (Question $\rightarrow$ Answer) teaches the model the *format*. Chain-of-Thought ICL (Question $\rightarrow$ Step-by-Step Logic $\rightarrow$ Answer) teaches the model the *reasoning*. 
*   **Action:** If the HuggingFace datasets contain the CoT rationales, we should inject them into the Teacher's 4 examples. The Teacher's accuracy will skyrocket to near 100%. Then, when we distill the Teacher's hidden states down to the 0-shot Student (which has no CoT), we will be directly distilling pure reasoning logic into the student's attention heads!
