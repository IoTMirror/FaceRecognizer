using Emgu.CV;

namespace FRLib
{
    public interface ImageProcessor
    {
        Mat Process(Mat image);
    }
}
