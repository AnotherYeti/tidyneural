#include "xor.h"
#include <cmath>

using namespace Neural;

xorActor::xorActor()
{
	fitness = 0.0f;
}

float xorActor::getFitness()
{
	return fitness;
}

void xorActor::test()
{
	float sumDistance = 0.0f;

	//0,0
	network->setInput(0, 1.0f);
	network->setInput(1, 0.0f);
	network->setInput(2, 0.0f);

	int netSteps = 10;
	for(int i = 0; i < netSteps; i++)
		network->step();	

	sumDistance += fabs(0.0f - network->getOutput(0));


	//1,1
	network->setInput(0, 1.0f);
	network->setInput(1, 1.0f);
	network->setInput(2, 1.0f);

	for(int i = 0; i < netSteps; i++)
		network->step();	

	sumDistance += fabs(0.0f - network->getOutput(0));

	//1,0
	network->setInput(0, 1.0f);
	network->setInput(1, 1.0f);
	network->setInput(2, 0.0f);

	for(int i = 0; i < netSteps; i++)
		network->step();	

	sumDistance += fabs(1.0f - network->getOutput(0));

	//0,1
	network->setInput(0, 1.0f);
	network->setInput(1, 0.0f);
	network->setInput(2, 1.0f);

	for(int i = 0; i < netSteps; i++)
		network->step();	

	sumDistance += fabs(1.0f - network->getOutput(0));

	sumDistance = 4.0f - sumDistance;
	sumDistance = sumDistance * sumDistance;

	fitness = sumDistance;
}

void xorActor::reset() 
{
	fitness = 0.0f;
}
