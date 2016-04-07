using System.Collections.Generic;
using Emgu.CV;

namespace FRLib
{
    public interface UserFaceRecognizer
    {
        int recognize(Mat image);
        int recognize(Mat image, List<int> usersToCheck);
        void train(int userId, Mat image);
        void removeUserData(int userId);
    }
}
