import m from 'mithril';
import '../css/login.scss';

const InputGroup = {
  view(vnode) {
    return m('div.input-group', [
      m('label', vnode.attrs.title),
      m(`input[type=${vnode.attrs.inType || 'text'}].auth-in`, {
        oninput: vnode.attrs.oninput,
      }),
    ]);
  }
}

export default class Login {
  constructor(vnode) {
    this.valid = false;
    this.signup = true;

    this.res = {
      firstName: '',
      lastName: '',
      email: '',
      password: '',
      confirmPassword: '',
    }
  }

  authenticate() {
    console.log(this.signup ? 'createUser' : 'authenticateUser');

    m.request(`http://localhost:2002/${this.signup ? 'createUser' : 'authenticateUser'}`, {
      method: 'POST',
      params: this.res,
    })
      .then(res => {
        if (res.uid)
          console.log(res.uid);
      })
      .catch(window.alert)
  }

  validate() {
    this.valid = this.res.firstName && this.res.lastName && this.res.email && 
                 this.res.email.includes('@') && this.res.email.includes('.') &&
                 this.res.password && this.res.confirmPassword && 
                 this.res.password === this.res.confirmPassword;
  }

  view(vnode) {
    return m('div.auth-popup', 
      m('div.auth-content', [
        this.signup ? m('div.hstack', [

          m(InputGroup, {
            title: 'First Name', 
            oninput: e => {
              this.res.firstName = e.target.value;
              this.validate();
            },
          }),

          m(InputGroup, {
            title: 'Last Name', 
            oninput: e => {
              this.res.lastName = e.target.value;
              this.validate();
            },
          }),

        ]) : '',

        m(InputGroup, {
          title: 'Email Address',
          oninput: e => {
            this.res.email = e.target.value;
            this.validate();
          },
        }),

        m(InputGroup, {
          title: 'Password',
          inType: 'password',
          oninput: e => {
            this.res.password = e.target.value;
            this.validate();
          },
        }),

        this.signup ? m(InputGroup, {
          title: 'Confirm Password',
          inType: 'password',
          oninput: e => {
            this.res.confirmPassword = e.target.value;
            this.validate();
          },
        }) : '',

        m('button.login-button', {
          disabled: !this.valid,
          onclick: e => {
            this.authenticate();
          }
        }, this.signup ? 'Sign Up' : 'Log In'),
      ])
    );
  }
}