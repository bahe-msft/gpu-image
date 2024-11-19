# tf nightly-gpu: https://hub.docker.com/layers/tensorflow/tensorflow/nightly-gpu/images/sha256-31e8f7fa6e7abee03c407deb776d272bac884b9f773e68161aae02d3a22932cb?context=explore
FROM tensorflow/tensorflow@sha256:31e8f7fa6e7abee03c407deb776d272bac884b9f773e68161aae02d3a22932cb

ADD model.py /opt/model.py

ENTRYPOINT ["python3", "/opt/model.py"]
