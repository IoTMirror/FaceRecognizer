using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;


namespace FRLib
{
    public interface Detector
    {
        List<Rectangle> Detect(Mat image);

        List<Mat> Extract(Mat image);

        Mat ExtractLargest(Mat image);
    }

}
