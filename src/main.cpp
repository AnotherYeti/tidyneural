#include "Neural.h"
#include "Actor.h"
#include <stdio.h>
#include <list>
#include "xor.h"
#include "Population.h"

using namespace Neural;
using namespace NEAT;
using namespace std;

void testBasicNet()
{
	//Simple simple network
	vector<NWeight> weights;
	weights.push_back(NWeight(0, 1, 0, 4.0f, true));
	vector<int> test;
	NNetwork* net = new NNetwork(&weights, 1, 1, &test);

	//Starting info
	printf("Starting state:\n");
	net->setInput(0,0.0f);
	printf("In: 0.0\n");
	printf("Out: %f\n", net->getOutput(0));

	net->setInput(0,1.0f);
	net->step();
	net->step();
	net->step();
	printf("Out: %f\n", net->getOutput(0));

	delete net;
}

void testPopulation()
{
	list<xorActor*> actors;
	for(int i = 0; i < 150; i++)
		actors.push_back(new xorActor());
	Population* pop = new Population((list<Actor*>*) &actors);
	pop->weightMutationRate = 0.80f;
	pop->weightMutationIntensity = 0.05f;
	pop->addWeightMutationRate = 0.05f;
	pop->addNodeMutationRate = 0.03f;
	pop->c1 = 1.0f;
	pop->c3 = 0.4f;
	pop->speciationDifference = 1.0f;

	for(int i = 0; i < 100; i++) {
		for(list<xorActor*>::iterator it = actors.begin();
																it != actors.end();
																it++) {
			if((*it)->hasNetwork())
				(*it)->test();
		}
		pop->stepGeneration(true);
		printf("\n");
	}

	delete pop;
	list<xorActor*>::iterator it;
	for(it = actors.begin(); it != actors.end(); it++)
		delete(*it);
}

void testXOR()
{
}

int main ()
{
	testBasicNet();
	testPopulation();
	testXOR();

	return 0;
}
