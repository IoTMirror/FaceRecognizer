using System.Collections.Generic;
using Emgu.CV;
using System.Drawing;

namespace FRLib
{
    public class CascadeDetector : Detector
    {
        public CascadeClassifier cascadeClassifier { get; set; }
        public double scaleFactor { get; set; } = 1.1;
        public int minNeighbours { get; set; } = 6;
        public Size minSize { get; set; } = default(Size);
        public Size maxSize { get; set; } = default(Size);

        public CascadeDetector(CascadeClassifier cascadeClassifier)
        {
            this.cascadeClassifier = cascadeClassifier;
        }

        public List<Rectangle> Detect(Mat image)
        {
            if(image==null) return new List<Rectangle>();
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayImage, grayImage);
            Rectangle[] facesArray = (cascadeClassifier.DetectMultiScale(grayImage, scaleFactor, minNeighbours, minSize, maxSize));
            if (facesArray.Length > 0) return new List<Rectangle>(facesArray);
            else return new List<Rectangle>();
        }

        public List<Mat> Extract(Mat image)
        {
            List<Mat> extractedFaces = new List<Mat>();
            List<Rectangle> faces = Detect(image);
            foreach (var roi in faces)
            {
                extractedFaces.Add(new Mat(image, roi));
            }
            return extractedFaces;
        }

        public Mat ExtractLargest(Mat image)
        {
            List<Mat> extractedFaces = Extract(image);
            if (extractedFaces.Count != 0)
            {
                int largestI = 0;
                int largestSize = extractedFaces[0].Size.Height * extractedFaces[0].Size.Width;
                for (int currentI=1;currentI<extractedFaces.Count;++currentI)
                {
                    int currentSize = extractedFaces[currentI].Size.Height * extractedFaces[currentI].Size.Width;
                    if (currentSize>largestSize)
                    {
                        largestI = currentI;
                        largestSize = currentSize;
                    }
                }
                return extractedFaces[largestI];
            }
            else return null;
        }
    }
}
