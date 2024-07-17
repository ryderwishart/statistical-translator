<!-- App.svelte -->
<script lang="ts">
	import { onMount } from 'svelte';
	import debounce from 'lodash/debounce';
	import { fetchSourceSentences, fetchGlosses } from '$lib/apiFunctions';

	interface GlossResult {
		source: string;
		target: string;
		glosses: {
			source_word: string;
			target_word: string;
			alignment: number[];
		}[];
	}

	let sourceSentences: string[] = [];
	let targetSentences: string[] = [];
	let glossResults: GlossResult[] = [];
	let highlightedSourceIndex: number | null = null;
	let highlightedTargetIndex: number | null = null;
	let error: string | null = null;

	onMount(async () => {
		try {
			sourceSentences = await fetchSourceSentences();
			targetSentences = new Array(sourceSentences.length).fill('');
		} catch (e) {
			console.error('Error loading source sentences:', e);
			error = 'Failed to load source sentences. Please try refreshing the page.';
		}
	});

	const updateGlosses = debounce(async () => {
		try {
			glossResults = await fetchGlosses(sourceSentences, targetSentences);
		} catch (e) {
			console.error('Error updating glosses:', e);
			error = 'Failed to update glosses. Please try again.';
		}
	}, 500);

	function handleInput(index: number) {
		return (event: Event) => {
			const target = event.target as HTMLTextAreaElement;
			targetSentences[index] = target.value;
			updateGlosses();
		};
	}

	function highlightAlignment(gloss: GlossResult['glosses'][0]) {
		highlightedSourceIndex = gloss.alignment[0];
		highlightedTargetIndex = gloss.alignment[1];
	}

	function clearHighlight() {
		highlightedSourceIndex = null;
		highlightedTargetIndex = null;
	}
</script>

<main class="container mx-auto p-4">
	{#if error}
		<div
			class="relative mb-4 rounded border border-red-400 bg-red-100 px-4 py-3 text-red-700"
			role="alert"
		>
			<strong class="font-bold">Error:</strong>
			<span class="block sm:inline">{error}</span>
		</div>
	{/if}

	<div class="grid grid-cols-2 gap-4">
		<div class="source-area">
			{#each sourceSentences as sentence, i}
				<div class="mb-4 rounded-lg bg-white p-4 shadow-md">
					<div class="source">
						{#each sentence.split(' ') as word, j}
							<span class="mr-1 inline-block {highlightedSourceIndex === j ? 'bg-yellow-200' : ''}"
								>{word}</span
							>
						{/each}
					</div>
				</div>
			{/each}
		</div>
		<div class="target-area">
			{#each targetSentences as sentence, i}
				<div class="mb-4 rounded-lg bg-white p-4 shadow-md">
					<textarea
						class="w-full rounded border p-2"
						value={sentence}
						on:input={handleInput(i)}
						placeholder="Enter translation here..."
					/>
				</div>
			{/each}
		</div>
	</div>

	<div class="gloss-area mt-8">
		<h2 class="mb-4 text-2xl font-bold">Glosses</h2>
		{#each glossResults as result, i}
			<div class="mb-4 rounded-lg bg-white p-4 shadow-md">
				<div class="gloss-sentence">
					{#each result.glosses as gloss, j}
						<span
							class="mr-2 inline-block cursor-pointer {highlightedTargetIndex === j
								? 'bg-yellow-200'
								: ''}"
							on:mouseover={() => highlightAlignment(gloss)}
							on:mouseout={clearHighlight}
						>
							{gloss.target_word}
						</span>
					{/each}
				</div>
			</div>
		{/each}
	</div>
</main>

<style>
	:global(body) {
		background-color: #f0f0f0;
	}
</style>
