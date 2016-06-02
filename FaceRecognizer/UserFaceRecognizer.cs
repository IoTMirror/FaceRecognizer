using System.Collections.Generic;
using Emgu.CV;

namespace FRLib
{
    public interface UserFaceRecognizer
    {
        int Recognize(Mat image);
        int Recognize(Mat image, List<int> usersToCheck);
        void Train(int userId, Mat image);
        void RemoveUserData(int userId);
    }
}
