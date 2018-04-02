using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.Math.Distribution;

namespace SimuKit.ML.Ensemble
{
    public class RandomEngine
    {
        public static int NextInt(int upper)
        {
            return DistributionModel.NextInt(upper);
        }
    }
}
