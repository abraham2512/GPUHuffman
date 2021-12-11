#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "parallelHeader.h"
#define block_size 1024
#define MIN_SCRATCH_SIZE 50 * 1024 * 1024

struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
unsigned char bitSequenceConstMemory[256][255];
struct huffmanDictionary huffmanDictionary;
unsigned int constMemoryFlag = 0;

int main(int argc, char **argv){
	unsigned int i;
	unsigned int distinctCharacters, combinedHuffmanNodes, inputFileLen, frequency[256];
	unsigned char *inputFileData, bitSequenceLength = 0, bitSequence[255];
	unsigned int *compressedDataOffset, cpu_time_used;
	unsigned int integerOverflowFlag;
	FILE *inputFile, *compressedFile;
	long unsigned int mem_free, mem_total;
	long unsigned int mem_req, mem_offset, mem_data;
	int numKernelRuns;
	clock_t start, end;
	
	// check number of args
	if(argc != 3){
		printf("try with arguments InputFile and OutputFile");
		return -1;
	}
	// read input file, get inputFileLen and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLen = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLen * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLen, inputFile);
	fclose(inputFile);
	
	// calculate run duration
	start = clock();
	
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < inputFileLen; i++){
		frequency[inputFileData[i]]++;
	}

	// initialize nodes of huffman tree
	distinctCharacters = 0;
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
			huffmanTreeNode[distinctCharacters].count = frequency[i];
			huffmanTreeNode[distinctCharacters].letter = i;
			huffmanTreeNode[distinctCharacters].left = NULL;
			huffmanTreeNode[distinctCharacters].right = NULL;
			distinctCharacters++;
		}
	}
	
	// build tree 
	for (i = 0; i < distinctCharacters - 1; i++){
		combinedHuffmanNodes = 2 * i;
		sortHuffmanTree(i, distinctCharacters, combinedHuffmanNodes);
		buildHuffmanTree(i, distinctCharacters, combinedHuffmanNodes);
	}
	
	if(distinctCharacters == 1){
	  head_huffmanTreeNode = &huffmanTreeNode[0];        
        }

	// build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);
	
	// calculate memory requirements
	// GPU memory
	cudaMemGetInfo(&mem_free, &mem_total);
	
	// debug
	if(1){
		printf("Free Mem: %lu\n", mem_free);		
	}

	// offset array requirements
	mem_offset = 0;
	for(i = 0; i < 256; i++){
		mem_offset += frequency[i] * huffmanDictionary.bitSequenceLength[i];
	}
	mem_offset = mem_offset % 8 == 0 ? mem_offset : mem_offset + 8 - mem_offset % 8;
	
	// other memory requirements
	mem_data = inputFileLen + (inputFileLen + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);
	
	if(mem_free - mem_data < MIN_SCRATCH_SIZE){
		printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, mem_data + MIN_SCRATCH_SIZE);
		return -1;
	}
	mem_req = mem_free - mem_data - 10 * 1024 * 1024;
	numKernelRuns = ceil((double)mem_offset / mem_req);
	integerOverflowFlag = mem_req + 255 <= UINT_MAX || mem_offset + 255 <= UINT_MAX ? 0 : 1;

	// debug
	if(1){
	printf("	InputFileSize      =%u\n\
	OutputSize         =%u\n\
	NumberOfKernel     =%d\n\
	integerOverflowFlag=%d\n", inputFileLen, mem_offset/8, numKernelRuns, integerOverflowFlag);		
	}

	
	// generate data offset array
	compressedDataOffset = (unsigned int *)malloc((inputFileLen + 1) * sizeof(unsigned int));

	// launch kernel
	lauchCUDAHuffmanCompress(inputFileData, compressedDataOffset, inputFileLen, numKernelRuns, integerOverflowFlag, mem_req);

	// calculate run duration
	end = clock();
	
	// write src inputFileLen, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	fwrite(&inputFileLen, sizeof(unsigned int), 1, compressedFile);
	fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
	fwrite(inputFileData, sizeof(unsigned char), mem_offset / 8, compressedFile);
	fclose(compressedFile);	
	
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	free(inputFileData);
	free(compressedDataOffset);
	return 0;
}
