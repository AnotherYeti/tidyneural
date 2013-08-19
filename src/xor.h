#ifndef XOR_H
#define XOR_H

#include "Actor.h"

class xorActor : public Neural::Actor
{
	public:
		xorActor();

		int getSensors() { return 3; }
		int getActions() { return 1; }
		
		float getFitness();
		void reset();

		void test();

	private:
		float fitness;
};

#endif
