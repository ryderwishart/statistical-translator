<!-- App.svelte -->
<script lang="ts">
	import { onMount } from 'svelte';
	import debounce from 'lodash/debounce';

	interface GlossResult {
		source: string;
		target: string;
		glosses: {
			source_word: string;
			target_word: string;
			alignment: number[];
		}[];
	}
	let sourceSentences: string[] = [
		'I am a sentence.',
		'I am another sentence.'
	];
	let targetSentences: string[] = [];
	let glossResults: GlossResult[] = [];

	onMount(async () => {
		// Load initial source sentences
		const response = await fetch('/api/source-sentences');
		// sourceSentences = await response.json();
		targetSentences = new Array(sourceSentences.length).fill('');
	});

	const updateGlosses = debounce(async () => {
		const response = await fetch('/api/gloss', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ source_sentences: sourceSentences, target_sentences: targetSentences })
		});
		glossResults = await response.json();
	}, 500);

	function handleInput(index) {
		return (event) => {
			targetSentences[index] = event.target.value;
			updateGlosses();
		};
	}

	function highlightAlignment(gloss) {
		// Implement highlighting logic here
	}
</script>

<main>
	<div class="translation-area">
		{#each sourceSentences as sentence, i}
			<div class="sentence-pair">
				<div class="source">{sentence}</div>
				<textarea class="target" value={targetSentences[i]} on:input={handleInput(i)}></textarea>
			</div>
		{/each}
	</div>

	<div class="gloss-area">
		{#each glossResults as result, i}
			<div class="gloss-sentence">
				{#each result.glosses as gloss}
					<span class="gloss-word" on:mouseover={() => highlightAlignment(gloss)}>
						{gloss.target_word}
					</span>
				{/each}
			</div>
		{/each}
	</div>
</main>

<style>
	/* Add your styles here */
</style>
