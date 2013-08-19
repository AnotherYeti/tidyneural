#ifndef NWEIGHT_H
#define NWEIGHT_H

namespace Neural 
{
	class NWeight
	{
		public:
			NWeight(int in, int out, int innovation, float weight, bool enabled);

			int getIn() const {return in;};
			int getOut() const {return out;};
			int getInnovation() const {return innovation;};
			bool isEnabled() const { return enabled;};
			float getWeight() const { return weight;};
			void setEnabled(bool enabled) { this->enabled = enabled; };
			
		private:
			int in;
			int out;
			int innovation;
			float weight;
			bool enabled;
	};
}

#endif
