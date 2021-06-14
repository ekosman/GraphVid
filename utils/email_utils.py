import smtplib
import ssl
import sys


class GmailNotifier:
    def __init__(self, username, password, to):
        self.username = username
        self.password = password
        self.to = to

    def __enter__(self):
        return self

    def send_args_description(self, subject, args):
        message = "Running command:\n"
        message += ' '.join(sys.argv)
        message += """
    ====== All settings used ======:\n
    """
        for k, v in sorted(vars(args).items()):
            message += f"      {k}: {v}\n"

        self.send(subject, message)

    def send_results(self, subject, args, results):
        message = "Running command:\n"
        message += ' '.join(sys.argv)
        message += """
        ====== All settings used ======:\n
        """
        for k, v in sorted(vars(args).items()):
            message += f"      {k}: {v}\n\n\n\n"

        message += """
        ====== All Results ======:\n
        """
        for k, v in results.items():
            message += f"      {k}: {v}\n\n\n\n"

        self.send(subject, message)

    def send(self, subject, message):
        port = 465
        context = ssl.create_default_context()
        my_email = f"{self.username}@gmail.com"
        message = 'Subject: {}\n\n{}'.format(subject, message)
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(my_email, self.password)
            server.sendmail(my_email, self.to, message)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
