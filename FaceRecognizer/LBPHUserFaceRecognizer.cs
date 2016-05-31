using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Util;
using System.IO;

namespace FRLib
{
    public class LBPHUserFaceRecognizer : UserFaceRecognizer
    {
        public string dataDirectory { get; set; }
        public int radius { get; set; } = 1;
        public int neighbours { get; set; } = 8;
        public int gridX { get; set; } = 8;
        public int gridY { get; set; } = 8;
        public double threshold { get; set; } = Double.MaxValue;

        public LBPHUserFaceRecognizer(string dataDirectory)
        {
            this.dataDirectory = dataDirectory;
        }

        public int recognize(Mat image)
        {
            if (image == null) return -1;
            IEnumerable<string> files = Directory.EnumerateFiles(dataDirectory, "*.lbph");
            List<int> users = new List<int>();
            foreach(string file in files)
            {
                string[] nameParts = file.Replace(".lbph", "").Split(new char[] { '/','\\' });
                int userid;
                if (Int32.TryParse(nameParts[nameParts.Length-1], out userid)) users.Add(userid);
            }
            return recognize(image, users);
        }

        public int recognize(Mat image, List<int> usersToCheck)
        {
            if (image == null) return -1;
            int bestUID = -1;
            double bestDistance=0;
            foreach(int uid in usersToCheck)
            {
                LBPHFaceRecognizer lbph = new LBPHFaceRecognizer(radius,neighbours,gridX,gridY,threshold);
                try
                {
                    lbph.Load(dataDirectory + "/" + uid + ".lbph");
                }
                catch (CvException)
                {
                    continue;
                }
                FaceRecognizer.PredictionResult result = lbph.Predict(image);
                if(result.Distance!=Double.MaxValue && (bestUID == -1 || result.Distance < bestDistance))
                {
                    bestUID = result.Label;
                    bestDistance = result.Distance;
                }
            }
            return bestUID;
        }

        public void removeUserData(int userId)
        {
            File.Delete(dataDirectory + "/" + userId + ".lbph");
        }

        public void train(int userId, Mat image)
        {
            LBPHFaceRecognizer lbph =  new LBPHFaceRecognizer(radius, neighbours, gridX, gridY, threshold);
            try
            {
                lbph.Load(dataDirectory + "/" + userId + ".lbph");
            }
            catch(CvException)
            {
                //nothing to do - new Face Recognizer
            }
            VectorOfMat images = new VectorOfMat();
            images.Push(image);
            VectorOfInt labels = new VectorOfInt();
            labels.Push(new int[] { userId });
            lbph.Update(images,labels);
            lbph.Save(dataDirectory + "/" +userId + ".lbph");
        }
    }
}
