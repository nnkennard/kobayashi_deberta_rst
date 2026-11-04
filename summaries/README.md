# Incorporating summaries into RST-DT parsing

## Training

For training, we just need summaries of the gold spans (inference is greedy... TODO: think about this).
We first need to extract all the spans that occur in gold trees in RST-DT.

Next, we use OLlama to summarize the EDUs in the spans. We choose to summarize any span that contains 20 or more (whitespace-delimited) words. The OLlama prompt requests that the summary of the span is one sentence in 20 or fewer words.

Finally, we gather the relations and summaries for analysis.
