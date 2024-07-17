# Training Models

Bu bölümde machine learning modellerinin nasıl eğitildiklerini göreceğiz. Aslında bu kısımı bilmeden de framework'leri kullanarak modellerinizi eğitip optimize edebilirsiniz. Ancak neyin nasıl çalıştığını bildiğinizde modelleri geliştirmek daha kolay oluyor. Çünkü bu alanda çalışacaksanız şunu anlıyorsunuz, denenecek çok fazla model ve yapılacak birçok deney var. Bunların hepsini yapmak büyük bir zaman kaybı olacaktır. Eğer arkadaki çalışma prensiplerini anlarsanız probleminize ve veri kümenizin özelliklerine göre hangi deneyleri yapacağınızı belirleyebilirsiniz. Bu da size zaman kazandırıp projelerinizi daha iyi yönetme kabiliyeti sağlayacaktır.

Bu bölümde basit çalışma prensibine sahip machine learning modellerinin arkasındaki sistemle başlayıp şuanki kullanılan matematiksel hesaplama yöntemlerine doğru gideceğiz. İlk olarak __Gradient Descent (GD)__'ile nasıl hesaplamalar yapıldığını öğreneceğiz. Ardından __Batch Gradient Descent__, __Mini-batch GD__, ve __Stochastic GD__ konularına bakacağız. Daha sonrasında birkaç __Regularization__ yöntemini öğrenip nasıl overfitting'i engelliyorlar onu göreceğiz. Son olarak __Linear Regression__, __Polynomial Regression__, __Logistic Regression__ ve __Softmax Regression__ modellerine göz atacağız.
